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
            "Your task is to write a cohesive, continuous master script for an entire presentation based on the raw speaker notes.\n"
            "CRITICAL RULES:\n"
            "1. FLOW: The script must flow naturally from one slide to the next (e.g. use transitions like 'Moving on to...', 'As we just saw...').\n"
            "2. EXPRESSIONS: You are ENCOURAGED to use inline expression tags natively supported by OmniVoice: [laughter], [sigh], [breath], [gasps], [clears throat].\n"
            "3. INSTRUCT: For each slide, write a custom voice `instruct` string to guide the AI Voice model. It MUST start with the speaker's gender and age, followed by emotional descriptors (e.g. 'female, young adult, highly excited, fast pacing', or 'male, middle-aged, whispering calmly').\n"
            "4. OUTPUT FORMAT: You MUST return a strictly valid JSON array of objects. Each object must have EXACTLY three keys:\n"
            "   - 'slide_number' (integer): The slide number.\n"
            "   - 'script_text' (string): The refined, natural speech for that slide (including inline expression tags).\n"
            "   - 'instruct' (string): The custom voice design instruct string for this segment."
        )

    async def generate_master_script(self, presentation, voice_gender: str, voice_age: int, model_name: str = "gpt-4o") -> dict:
        """Refines all speaker notes into a continuous flowing script and custom voice instructions."""
        logger.info("generating_master_script", total_slides=len(presentation.slides))
        
        user_message = f"Base Voice Profile: {voice_gender}, Age {voice_age}\n\n"
        for slide in presentation.slides:
            user_message += f"Slide {slide.slide_number}: {slide.title}\nRaw Notes: {slide.speaker_notes}\n\n"
            
        user_message += "Rewrite these into a continuous, highly engaging avatar script with natural transitions between slides. Output a JSON array."
        
        request = ChatRequest(
            model=model_name,
            messages=[
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(role="user", content=user_message)
            ],
            temperature=0.7,
            max_tokens=2048
        )
        
        try:
            response = await self.llm_client.complete(request)
            parsed_array = json_repair.repair_json(response.content, return_objects=True)
            
            # Convert array to a dictionary mapped by slide_number for easy access
            script_dict = {}
            if isinstance(parsed_array, list):
                for item in parsed_array:
                    s_num = item.get("slide_number")
                    if s_num is not None:
                        script_dict[s_num] = {
                            "script_text": item.get("script_text", ""),
                            "instruct": item.get("instruct", f"{voice_gender}, clear articulation")
                        }
            return script_dict
        except Exception as e:
            logger.error("master_script_generation_failed", error=str(e))
            # Fallback
            script_dict = {}
            for slide in presentation.slides:
                script_dict[slide.slide_number] = {
                    "script_text": slide.speaker_notes,
                    "instruct": f"{voice_gender}, professional"
                }
            return script_dict
