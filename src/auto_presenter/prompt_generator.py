import structlog
from src.content_engine.llm.client import LLMClient
from src.content_engine.llm.schemas import ChatRequest, ChatMessage

logger = structlog.get_logger(__name__)

class PromptGenerator:
    """Generates highly visual, 3D premium tech style prompts for Z-Image Turbo."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        
        self.system_prompt = (
            "You are an expert prompt engineer specializing in generating prompts for Z-Image-Turbo. "
            "Your goal is to generate prompts that create stunning, highly informative presentation slides in the EXACT visual style of NotebookLM infographics. "
            "Follow these STRICT rules:\n"
            "1. Aesthetic (CRITICAL): Always start the prompt with: 'NotebookLM visual style, highly detailed infographic. Flat 2D vector line-art, strict isometric projection.'\n"
            "2. Background & Framing: Explicitly require: 'A solid, clean cream background with absolutely NO grid lines and NO graph paper. The content is framed by minimalist UI wireframe boxes and sharp, thin black borders, resembling a clean digital workspace.'\n"
            "3. Color Palette: Enforce the NotebookLM palette: 'Crisp white, deep black, light blue, and vibrant orange accents.'\n"
            "4. Visual Storytelling & Density (CRITICAL): The slide MUST have high visual density and tell a comprehensive story. DO NOT create sparse, empty diagrams (like a simple timeline with just laptops and dates). Use the wireframe panels to group multiple interacting components (e.g., 'Legacy Mainframes' vs 'Modern Containers').\n"
            "5. Zero Spelling Errors (CRITICAL): Z-Image-Turbo fails on long sentences. You MUST simplify all text into extremely short, common 2-3 word labels (e.g., 'Dev Laptop', 'Cloud Server', 'Legacy System'). However, these labels MUST be descriptive enough to explain the concept (e.g., don't just label '1970s', label it 'Legacy Mainframes'). DO NOT ask for full sentences.\n\n"
            "You will be given a list of slides. You MUST return a STRICT JSON array of strings, where each string is the prompt for the corresponding slide in order. Do not wrap it in any other objects.\n\n"
            "Example Input:\n"
            "Slide 1: Dependency Hell\n"
            "Text: The Pain: Modern apps require complex specific configs.\n"
            "Concept: A laptop connected to a server with tangled red wires.\n\n"
            "Example Output:\n"
            "[\"NotebookLM visual style, highly detailed infographic. Flat 2D vector line-art, strict isometric projection. A solid, clean cream background with absolutely NO grid lines. The content is framed by minimalist UI wireframe boxes and sharp, thin black borders, resembling a clean digital workspace. Color palette: crisp white, deep black, light blue, and vibrant orange accents. A large, bold monospace title at the top says exactly: \\\"The Problem\\\". Visual Storytelling layout: On the left, inside a wireframe box labeled exactly \\\"Dev Laptop\\\", sits an isometric laptop. On the right, inside a wireframe box labeled exactly \\\"Cloud Server\\\", sits an isometric server rack. Between them is a massive, chaotic tangle of bright red wires. A floating orange label on the wires says exactly: \\\"Dependency Hell\\\". A sharp rectangular box at the bottom says exactly: \\\"Complex Configs\\\".\"]"
        )

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
