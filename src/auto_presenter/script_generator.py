import structlog
import json_repair
from src.content_engine.llm.client import LLMClient
from src.content_engine.llm.schemas import ChatRequest, ChatMessage

logger = structlog.get_logger(__name__)

class ScriptGenerator:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.system_prompt = (
            "You are an expert presentation scriptwriter and Voice Director.\n"
            "Your task is to take the provided raw 'speaker_notes' and rewrite them into a natural, "
            "highly engaging spoken script tailored for an AI avatar.\n"
            "CRITICAL RULES:\n"
            "1. EXPRESSIONS: You MUST NOT use any non-verbal expression tags (like [laughter], [sigh], etc). SadTalker cannot animate them properly. Generate pure speech only.\n"
            "2. PACING: Keep sentences short and conversational.\n"
            "3. EMOTION: Determine the overall emotional tone of this slide. You MUST strictly output 'Neutral'. (Other emotions cause audio artifacts for this specific avatar).\n"
            "4. Output format: You MUST return a strictly valid JSON object with EXACTLY two keys:\n"
            "   - 'script_text' (string): The refined text (pure speech, NO tags).\n"
            "   - 'emotion' (string): The selected emotional tone."
        )

    async def generate_script(self, speaker_notes: str, model_name: str = "gpt-4o") -> dict:
        """Refines speaker notes into a VoiceX-compatible script and extracts the emotion."""
        logger.info("generating_refined_script")
        
        user_message = f"Raw Speaker Notes:\n{speaker_notes}\n\nRewrite this into an engaging avatar script."
        
        request = ChatRequest(
            model=model_name,
            messages=[
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(role="user", content=user_message)
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        try:
            response = await self.llm_client.complete(request)
            parsed_json = json_repair.repair_json(response.content, return_objects=True)
            return parsed_json
        except Exception as e:
            logger.error("script_generation_failed", error=str(e))
            # Fallback to the raw speaker notes if the LLM fails
            return {"script_text": speaker_notes, "emotion": "Neutral"}
