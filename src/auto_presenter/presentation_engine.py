import structlog
import json_repair
from typing import Optional
from pathlib import Path

from src.content_engine.llm.client import LLMClient
from src.content_engine.llm.schemas import ChatRequest, ChatMessage
from src.content_engine.orchestrator import run_content_engine
from .schemas import PresentationOutline

logger = structlog.get_logger(__name__)

class PresentationEngine:
    """Perfect Content Engine: Converts raw knowledge into a creative presentation outline."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        
        self.base_system_prompt = (
            "You are an expert presentation designer. Your task is to take the provided source context "
            "and convert it into a highly structured, engaging slide presentation.\n"
            "CRITICAL RULES:\n"
            "1. ZERO HALLUCINATION: You must ONLY use facts provided in the source context. Do not invent information.\n"
            "2. Creativity: Structure the presentation logically (Intro, Problem, Solution, Deep Dive, Conclusion).\n"
            "3. Bullet Points: Keep bullet points concise (max 4 per slide).\n"
            "4. Script: Write a natural, conversational speaker script for the avatar to read.\n"
            "5. You MUST return your response as a strictly valid JSON object with EXACTLY two root keys:\n"
            "   - 'topic' (string)\n"
            "   - 'slides' (array of objects containing slide_number, title, bullet_points, speaker_notes, visual_concept)."
        )

    async def generate_outline(self, topic: str, context: Optional[str] = None, model_name: str = "gpt-4o", num_slides: int = 10, strict_research: bool = False) -> PresentationOutline:
        """
        Generates a PresentationOutline based on the context. If context is missing, it falls back to either a fast LLM generation or exhaustive web research.
        num_slides: Controls the exact number of slides to generate (min 1, max 10).
        strict_research: If True, uses the exhaustive 5-minute ContentEngine web-search pipeline to guarantee zero-hallucination when no context is provided.
        """
        # Enforce min/max slides rule
        num_slides = max(1, min(10, num_slides))
        
        # Inject the dynamic slide count rule into the system prompt
        dynamic_system_prompt = self.base_system_prompt + f"\n6. Length: You MUST generate EXACTLY {num_slides} slides. No more, no less.\n"
        
        logger.info("generating_presentation_outline", topic=topic, has_user_context=bool(context), num_slides=num_slides, strict_research=strict_research)
        
        # Fallback: If no context is provided, figure out how to get it
        if not context or not context.strip():
            if strict_research:
                logger.info("no_context_provided_running_strict_research_pipeline")
                # We pass empty files list, relying on the orchestrator's exhaustive RAG/Search capabilities
                from src.content_engine.orchestrator import run_content_engine
                manuscript = await run_content_engine(topic=topic, uploaded_files=[], llm_client=self.llm_client)
                
                # Extract the raw markdown body from the synthesized manuscript to use as context
                context_blocks = []
                for section in manuscript.sections:
                    context_blocks.append(f"## {section.title}\n{section.body_markdown}")
                context = "\n\n".join(context_blocks)
                logger.info("strict_research_context_gathered", context_length=len(context))
            else:
                logger.info("no_context_provided_fast_fallback")
                
                context_request = ChatRequest(
                    model=model_name,
                    messages=[
                        ChatMessage(role="system", content="You are a knowledgeable expert. Write a comprehensive, highly detailed 5-paragraph educational summary about the given topic. This will be used as the foundational source material for a presentation."),
                        ChatMessage(role="user", content=f"Topic: {topic}")
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )
                
                context_response = await self.llm_client.complete(context_request)
                context = context_response.content
                logger.info("fast_fallback_context_generated", context_length=len(context))

        user_message = (
            f"Topic: {topic}\n\n"
            f"Source Context:\n{context}\n\n"
            "Based strictly on the source context above, generate a complete presentation outline. "
            "Return ONLY a JSON object."
        )
        
        request = ChatRequest(
            model=model_name,
            messages=[
                ChatMessage(role="system", content=dynamic_system_prompt),
                ChatMessage(role="user", content=user_message)
            ],
            temperature=0.3, # Low temp for zero-hallucination adherence
            max_tokens=4000,
            response_format={"type": "json_object"}
        )
        
        response = await self.llm_client.complete(request)
        
        # Parse the JSON response into our Pydantic model
        parsed_json = json_repair.repair_json(response.content, return_objects=True)
        
        # Robustly handle cases where the LLM incorrectly wraps the JSON in an outer key (e.g. {"presentation": {...}})
        if "topic" not in parsed_json and "slides" not in parsed_json:
            for key, value in parsed_json.items():
                if isinstance(value, dict) and ("topic" in value or "slides" in value):
                    parsed_json = value
                    break
            
        return PresentationOutline(**parsed_json)
