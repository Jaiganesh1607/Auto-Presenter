import structlog
from src.content_engine.llm.client import LLMClient
from src.content_engine.llm.schemas import ChatRequest, ChatMessage

logger = structlog.get_logger(__name__)

class PromptGenerator:
    """Generates highly visual, 3D premium tech style prompts for Z-Image Turbo."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        
        self.system_prompt = (
            "You are an expert prompt engineer specializing in generating prompts for Z-Image-Turbo (a state-of-the-art diffusion model). "
            "Your goal is to generate prompts that perfectly replicate the official NotebookLM presentation visual style, focusing specifically on how it brilliantly combines Text and Visuals. "
            "Follow these STRICT rules:\n"
            "1. Aesthetic (CRITICAL): The images MUST look like premium NotebookLM technical blueprints. Start every prompt with: 'NotebookLM visual style, highly detailed academic infographic.' Then, choose between 'Stunning 3D isometric technical schematic' OR 'Flat 2D vector architectural blueprint'.\n"
            "2. Background & Framing: Explicitly require: 'A clean cream/off-white background with a faint, subtle light-blue engineering graph-paper grid.'\n"
            "3. Text + Visual Layout (CRITICAL): You MUST structurally divide the canvas into a 'Text Zone' and a 'Visual Zone' (do not overlap them). For example: 'On the left side, large bold text... On the right side, a complex 3D diagram...' OR 'Across the top, a massive title banner... In the center below, a technical blueprint...'\n"
            "4. Z-Image Text Constraints: Z-Image-Turbo can render text if it is large, short, and well-placed. In the Text Zone, request exactly 1 massive bold Title (e.g., 'KUBERNETES') and 1 or 2 short, punchy subtitles/labels (e.g., 'Self-Healing Infrastructure'). NEVER ask for paragraphs.\n"
            "5. Visual Zone Metaphors: In the Visual Zone, describe the vibrant tech accents (cyan glass, magenta chips, golden platforms, or navy line-art) forming the core diagram that visually represents the text.\n"
            "6. Avatar Safe Zone (CRITICAL): The bottom-right quadrant of EVERY image MUST be completely empty (negative space). Explicitly add this to every prompt: 'The bottom-right corner is left completely empty as negative space with no objects or text.'\n\n"
            "You will be given a single slide's content. You MUST return ONLY the raw prompt string. Do not wrap it in JSON or any other objects.\n\n"
            "Example Input:\n"
            "Title: Dependency Hell\n"
            "Key Points: Modern apps require complex specific configs.\n"
            "Visual Metaphor / Concept: A laptop connected to a server with tangled red wires.\n\n"
            "Example Output:\n"
            "NotebookLM visual style, highly detailed academic infographic. Stunning 3D isometric technical schematic. A clean cream background with a faint, subtle light-blue engineering graph-paper grid. On the left side (Text Zone), a massive, bold navy-blue title says exactly: \"MULTI-CLOUD\". Below it, a clean subtitle says exactly: \"Strategic Architecture\". On the right side (Visual Zone), a beautiful 3D isometric diagram of floating translucent cyan glass servers connected to bright magenta processing chips sitting on golden-yellow platforms. Deep navy-blue architectural lines connect the components. The bottom-right corner is left completely empty as negative space with no objects or text."
        )

    async def generate_prompt_for_slide(self, slide, model_name: str = "gpt-4o") -> str:
        """Generates a highly creative prompt for a single slide to prevent repetitive batch patterns."""
        logger.info("generating_single_prompt", slide_number=slide.slide_number)
        
        content = "\n".join([f"- {b}" for b in slide.bullet_points])
        user_content = (
            "Generate a highly creative, specific prompt for the following presentation slide. "
            "Make sure to strictly follow the NotebookLM layout rules, keep text to a minimum, and output ONLY the raw prompt string.\n\n"
            f"Title: {slide.title}\n"
            f"Key Points:\n{content}\n"
            f"Visual Metaphor / Concept: {slide.visual_concept}\n"
        )
        
        request = ChatRequest(
            model=model_name,
            messages=[
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(role="user", content=user_content)
            ],
            temperature=0.8, # Increased temperature to force more creativity and variety between slides
            max_tokens=600
        )
        
        response = await self.llm_client.complete(request)
        
        # Clean up the output string
        prompt = response.content.strip().strip('"').strip("'")
        return prompt

    async def generate_prompts_batch(self, slides: list, model_name: str = "gpt-4o") -> list[str]:
        """Takes a list of SlideOutline objects, batches them, and returns a list of prompts via a single LLM call."""
        logger.info("generating_prompts_batch", total_slides=len(slides))
        
        # Build a consolidated prompt for all slides to minimize API calls
        user_content = "Generate a prompt for each of the following slides. Output ONLY a JSON array of strings.\n\n"
        for i, slide in enumerate(slides):
            content = "\n".join([f"- {b}" for b in slide.bullet_points])
            user_content += f"Slide {i+1}: {slide.title}\nBullets:\n{content}\nConcept: {slide.visual_concept}\n\n"
            
        request = ChatRequest(
            model=model_name,
            messages=[
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(role="user", content=user_content)
            ],
            temperature=0.7,
            max_tokens=2048 # Increased to allow for multiple prompts
        )
        
        response = await self.llm_client.complete(request)
        
        # Parse the JSON array
        import json_repair
        prompts = json_repair.repair_json(response.content, return_objects=True)
        
        # Ensure it's a flat list of strings
        if not isinstance(prompts, list):
            # Fallback if the LLM returned a dictionary (e.g. {"prompts": [...]})
            for key, value in prompts.items():
                if isinstance(value, list):
                    prompts = value
                    break
                    
        flat_prompts = []
        for item in prompts:
            if isinstance(item, list) and len(item) > 0:
                flat_prompts.append(str(item[0]))
            elif isinstance(item, str):
                flat_prompts.append(item)
            else:
                flat_prompts.append(str(item))
                
        return flat_prompts
