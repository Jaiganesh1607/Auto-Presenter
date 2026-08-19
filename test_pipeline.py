import asyncio
import os
import structlog
from pathlib import Path
from src.content_engine.llm.client import LLMClient
from src.content_engine.llm.providers import get_provider
from src.auto_presenter.prompt_generator import PromptGenerator
from src.auto_presenter.image_generator import ImageGeneratorClient
from src.auto_presenter.pipeline import PipelineOrchestrator
from src.auto_presenter.presentation_engine import PresentationEngine
from src.auto_presenter.script_generator import ScriptGenerator
from src.auto_presenter.audio_generator import VoiceXClient

logger = structlog.get_logger(__name__)

async def main():
    # Load configuration
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check if user set the Cloudflare URL
    if not os.environ.get("ZIMAGE_API_URL"):
        logger.error("missing_env_var", message="Please set ZIMAGE_API_URL in your .env file to your Cloudflare tunnel link (e.g., https://your-tunnel.trycloudflare.com).")
        return

    # 1. Initialize Engines
    provider = get_provider("nvidia")
    llm_client = LLMClient(provider=provider)
    presentation_engine = PresentationEngine(llm_client)
    prompt_gen = PromptGenerator(llm_client)
    image_client = ImageGeneratorClient()
    script_gen = ScriptGenerator(llm_client)
    audio_client = VoiceXClient()
    
    output_dir = Path("./output_images")
    orchestrator = PipelineOrchestrator(
        prompt_engine=prompt_gen, 
        image_client=image_client, 
        output_dir=output_dir,
        script_engine=script_gen,
        audio_client=audio_client
    )
    
    topic = "Docker and Containerization"
    model_name = "meta/llama-3.1-8b-instruct"
    
    # 2. Get the real outline via the Presentation Engine (Phase 1.5)
    logger.info("generating_presentation_outline", topic=topic)
    presentation = await presentation_engine.generate_outline(topic=topic, model_name=model_name, num_slides=3)
    
    # Save the intermediate presentation outline to output/
    meta_dir = Path("./output")
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "presentation_outline.json", "w") as f:
        f.write(presentation.model_dump_json(indent=4))
    
    # 3. Run the concurrent pipeline using the batched Prompt Engine & Image Queue
    voice_instruct = "female, professional, high pitch, clear articulation"
    await orchestrator.process_presentation(
        presentation=presentation, 
        voice_instruct=voice_instruct,
        model_name=model_name
    )
    
if __name__ == "__main__":
    asyncio.run(main())
